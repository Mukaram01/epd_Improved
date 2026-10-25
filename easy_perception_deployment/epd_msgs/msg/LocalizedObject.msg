string name # object name
sensor_msgs/RegionOfInterest roi # object 2d bounding box
sensor_msgs/Image segmented_binary_mask # object 2d segmentation mask
geometry_msgs/Point centroid  # object 3d bounding box centroid (camera frame)
float64 length  # object 3d bounding box length
float64 breadth  # object 3d bounding box breadth
float64 height  # object 3d bounding box height
sensor_msgs/PointCloud2 segmented_pcl # object 3d segmentation cloud
geometry_msgs/Vector3 axis # object major axis in camera frame (PCA on segmented_pcl)
geometry_msgs/Pose pose # pose in camera frame (+X along axis, +Z camera optical, +Y right-handed)
float64 confidence  # detector score of this object (0..1), as reported by the model
uint32 mask_pixel_count  # pixels in the object's segmentation mask
uint32 valid_depth_pixel_count  # mask pixels with a valid measured depth
float64 valid_depth_ratio  # valid_depth_pixel_count / mask_pixel_count
