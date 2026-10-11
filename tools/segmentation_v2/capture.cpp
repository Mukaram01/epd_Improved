// Disposable supervised acquisition only. No ROS, planning or controller APIs.
// Reuses the pinned typed-clear, depth-tested Ogre2 native MRT implementation.
#include "paired_native/native_pass.hh"
#include "physics_owner/identity_inventory.hh"
#include <ignition/gazebo/Server.hh>
#include <ignition/gazebo/ServerConfig.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/rendering/RenderUtil.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/rendering/ogre2/Ogre2Geometry.hh>
#include <ignition/rendering/ogre2/Ogre2Visual.hh>
#include <OgreItem.h>
#include <OgreSubItem.h>

namespace g=ignition::gazebo;
namespace c=g::components;
const char *pilotVS=R"(#version 330 core
in vec4 vertex; in vec3 normal;
uniform mat4 worldViewProj; uniform mat4 world;
out vec3 n;
void main(){gl_Position=worldViewProj*vertex;n=mat3(world)*normal;}
)";
const char *pilotFS=R"(#version 330 core
in vec3 n; uniform vec3 baseColour; uniform vec3 lightDirection;
uniform float ambient; uniform uint entityId;
layout(location=0) out vec4 rgb; layout(location=1) out uint identity;
void main(){float diffuse=max(dot(normalize(n),normalize(lightDirection)),0.0);
rgb=vec4(baseColour*(ambient+(1.0-ambient)*diffuse),1.0);identity=entityId;}
)";

struct Capture final:g::System,g::ISystemPostUpdate {
  g::RenderUtil render;
  Json::Value spec,report;
  std::string prefix;
  bool ready=false,done=false,ok=false;
  Capture(Json::Value s,std::string p):spec(s),prefix(p){}
  void PostUpdate(const g::UpdateInfo &info,const g::EntityComponentManager &ecm) override {
    if(done)return;
    try {
      if(!ready){render.SetEngineName("ogre2");render.SetSceneName("epd_pilot");
        render.SetHeadlessRendering(false);render.SetEnableSensors(false);render.Init();ready=true;}
      render.UpdateFromECM(info,ecm);render.Update();
      if(info.iterations!=2)return;
      done=true;
      auto scene=render.Scene();
      auto native=std::dynamic_pointer_cast<ignition::rendering::Ogre2Scene>(scene);
      if(!native)throw std::runtime_error("native Ogre2 required");
      auto sm=native->OgreSceneManager();
      Ogre::Root::getSingleton().getRenderSystem()->postExtraThreadsStarted();
      GLint major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
      if(major<4 || (major==4 && minor<5))throw std::runtime_error("OpenGL 4.5 required");
      auto &pm=Ogre::HighLevelGpuProgramManager::getSingleton();
      for(auto stage:{Ogre::GPT_VERTEX_PROGRAM,Ogre::GPT_FRAGMENT_PROGRAM}){
        auto p=pm.createProgram(stage==Ogre::GPT_VERTEX_PROGRAM?"pilot_vs":"pilot_fs","General","glsl",stage);
        p->setSource(stage==Ogre::GPT_VERTEX_PROGRAM?pilotVS:pilotFS);p->load();
        if(p->hasCompileError())throw std::runtime_error("shader compile failure");
      }
      auto inventory=workcell::IdentityInventory(ecm);
      report["inventory"]=inventory;
      std::set<Ogre::Item*> items;std::set<unsigned> ids;
      for(const auto &v:inventory["visuals"]){
        const auto visualId=v["id"].asUInt64();
        auto link=ecm.Component<c::ParentEntity>(visualId);
        auto model=link?ecm.Component<c::ParentEntity>(link->Data()):nullptr;
        auto name=model?ecm.Component<c::Name>(model->Data()):nullptr;
        if(!name || v["geometry"]["type"]!="BOX")throw std::runtime_error("incomplete ECM ownership");
        Json::Value authored;unsigned matches=0;
        for(const auto &obj:spec["objects"])if(obj["name"].asString()==name->Data()){authored=obj;++matches;}
        if(matches!=1)throw std::runtime_error("unknown/duplicate authored model");
        auto visual=render.SceneManager().VisualById(visualId);
        if(!visual || visual->GeometryCount()!=1)throw std::runtime_error("missing native visual");
        auto geo=std::dynamic_pointer_cast<ignition::rendering::Ogre2Geometry>(visual->GeometryByIndex(0));
        auto node=std::dynamic_pointer_cast<ignition::rendering::Ogre2Visual>(visual);
        auto item=geo?dynamic_cast<Ogre::Item*>(geo->OgreObject()):nullptr;
        unsigned id=visual->Id();
        if(!item || !node || geo->Parent()!=visual || item->getParentSceneNode()!=node->Node() ||
           item->getNumSubItems()!=1 || !items.insert(item).second || id==0 || id==UINT32_MAX || !ids.insert(id).second)
          throw std::runtime_error("ambiguous native renderer ownership");
        auto material=ecm.Component<c::Material>(visualId);
        if(!material)throw std::runtime_error("missing ECM material");
        auto colour=material->Data().Diffuse();
        auto mat=Ogre::MaterialManager::getSingleton().create("pilot_mat_"+std::to_string(id),"General");
        auto p=mat->getTechnique(0)->getPass(0);p->setVertexProgram("pilot_vs");p->setFragmentProgram("pilot_fs");
        p->getVertexProgramParameters()->setNamedAutoConstant("worldViewProj",Ogre::GpuProgramParameters::ACT_WORLDVIEWPROJ_MATRIX);
        p->getVertexProgramParameters()->setNamedAutoConstant("world",Ogre::GpuProgramParameters::ACT_WORLD_MATRIX);
        auto params=p->getFragmentProgramParameters();
        params->setNamedConstant("baseColour",Ogre::Vector3(colour.R(),colour.G(),colour.B()));
        const auto light=spec["light_direction"];
        params->setNamedConstant("lightDirection",Ogre::Vector3(light[0].asFloat(),light[1].asFloat(),light[2].asFloat()));
        params->setNamedConstant("ambient",spec["ambient"].asFloat());params->setNamedConstant("entityId",id);
        Ogre::HlmsMacroblock macro;macro.mDepthCheck=true;macro.mDepthWrite=true;p->setMacroblock(macro);
        Ogre::HlmsBlendblock blend;blend.mSourceBlendFactor=Ogre::SBF_ONE;blend.mDestBlendFactor=Ogre::SBF_ZERO;p->setBlendblock(blend);
        mat->load();item->setMaterial(mat);
        Json::Value row;row["renderer_id"]=id;row["visual_entity"]=Json::UInt64(visualId);
        row["link_entity"]=Json::UInt64(link->Data());row["model_entity"]=Json::UInt64(model->Data());
        row["model"]=name->Data();row["class_id"]=authored["class_id"];
        row["item_id"]=Json::UInt64(item->getId());report["mapping"].append(row);
      }
      auto actual=sm->getMovableObjectIterator("Item");unsigned count=0;
      while(actual.hasMoreElements()){++count;if(!items.count(dynamic_cast<Ogre::Item*>(actual.getNext())))throw std::runtime_error("unmapped Item");}
      if(count!=spec["objects"].size() || sm->getMovableObjectIterator("Entity").hasMoreElements())throw std::runtime_error("incomplete scene inventory");
      auto camera=sm->createCamera("pilot_camera");
      camera->setFixedYawAxis(true,Ogre::Vector3::UNIT_Z);camera->setPosition(0,-.40,.28);camera->lookAt(0,0,.025);
      camera->setNearClipDistance(.01);camera->setFarClipDistance(2);camera->setAspectRatio(1);camera->setFOVy(Ogre::Radian(1.0471975511965976));
      auto t256=PrepareNativePass(sm,camera,256);auto t512=PrepareNativePass(sm,camera,512);
      scene->PreRender();sm->updateSceneGraph();
      auto before=workcell::IdentityInventory(ecm);
      CaptureNativePass(t256,report,prefix,256,nullptr);
      CaptureNativePass(t512,report["native512"],prefix+".native512",512,nullptr);
      report["frozen_equal"]=before==workcell::IdentityInventory(ecm) && render.SimTime()==info.simTime;
      if(!report["frozen_equal"].asBool())throw std::runtime_error("scene changed during synchronous capture");
      scene->PostRender();
      auto cm=Ogre::Root::getSingleton().getCompositorManager2();cm->removeWorkspace(t256.workspace);cm->removeWorkspace(t512.workspace);
      report["acquisition"]["step"]=Json::UInt64(info.iterations);report["acquisition"]["stamp_ns"]=Json::Int64(info.simTime.count());
      report["acquisition"]["phase"]="blocking_PostUpdate_step2_two_native_passes_no_updates";
      report["rgb_profile"]="EPD_directional_diffuse_MRT_v1_no_shadows_no_MSAA";
      report["success"]=true;ok=true;
    }catch(const std::exception &e){done=true;report["success"]=false;report["failure"]=e.what();}
    if(done){std::ofstream out(prefix);out<<report<<'\n';out.flush();}
  }
};

int main(int argc,char **argv){
  if(argc!=4)return 2;
  for(const auto &s:{"",".rgb8",".ids.u32"})if(std::filesystem::exists(std::string(argv[3])+s))return 2;
  Json::Value spec;std::ifstream in(argv[2]);in>>spec;
  auto capture=std::make_shared<Capture>(spec,argv[3]);
  try{g::ServerConfig config;config.SetSdfFile(argv[1]);g::Server server(config);
    if(server.AddSystem(capture)!=std::optional<bool>(true))return 2;
    server.Run(true,2,false);
  }catch(const std::exception &e){capture->report["failure"]=e.what();capture->ok=false;}
  std::ofstream out(argv[3]);out<<capture->report<<'\n';return capture->ok?0:2;
}
