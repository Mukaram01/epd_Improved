"""Conservative admission estimates; historical full-run gate is unchanged."""
def project(policy, train_max, validation_max, peak_kib, free_bytes):
    stage=(1200*train_max+48*validation_max)*policy['feasibility_safety_factor']+policy['feasibility_overhead_seconds']
    aggregate=2*stage
    memory=peak_kib*policy['feasibility_safety_factor']
    disk=policy['disk_reserve_gib']*1024**3+8*400*1024**2
    return dict(estimated_stage_seconds=stage,estimated_aggregate_seconds=aggregate,
                projected_rss_kib=memory,required_disk_bytes=disk,
                feasible=stage<=policy['stage_timeout_seconds'] and aggregate<=policy['aggregate_timeout_seconds']
                and memory<=policy['memory_mib']*1024 and free_bytes>disk)


def remaining_admission(policy, history, validation_seconds, charged_seconds):
    remaining=(1200*max(r['seconds'] for r in history)+4*max(validation_seconds))*policy['feasibility_safety_factor']+policy['feasibility_overhead_seconds']
    return dict(estimated_remaining_seconds=remaining,charged_stage_seconds=charged_seconds,
                projected_aggregate_seconds=charged_seconds+remaining,
                remaining_aggregate_budget_seconds=policy['aggregate_timeout_seconds']-charged_seconds,
                admitted=remaining<=policy['stage_timeout_seconds'] and charged_seconds+remaining<=policy['aggregate_timeout_seconds'],
                method='1200 * observed maximum step wall +4 * observed maximum full validation epoch wall; unchanged1.5 factor +120s')
