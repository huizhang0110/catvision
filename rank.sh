

declare -A host_rank_map
host_rank_map["ucloud-wlcb-gpu-010"]=0
host_rank_map["ucloud-wlcb-gpu-011"]=1
host_rank_map["ucloud-wlcb-gpu-071"]=2
host_rank_map["ucloud-wlcb-gpu-072"]=3

host_rank_map["ucloud-wlcb-gpu-070"]=4
host_rank_map["ucloud-wlcb-gpu-074"]=5
host_rank_map["ucloud-wlcb-gpu-075"]=6
host_rank_map["ucloud-wlcb-gpu-077"]=7



input_host=$(hostname)

echo ${host_rank_map[$input_host]}

