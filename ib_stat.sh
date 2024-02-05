

declare -A host_rank_map
host_rank_map["ucloud-wlcb-gpu-010"]="=mlx5_0:1/IB,mlx5_1:1/IB,mlx5_2:1/IB,mlx5_3:1/IB"
host_rank_map["ucloud-wlcb-gpu-011"]="=mlx5_0:1/IB,mlx5_1:1/IB,mlx5_2:1/IB,mlx5_3:1/IB"
host_rank_map["ucloud-wlcb-gpu-070"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-071"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-072"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-073"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-074"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-075"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"
host_rank_map["ucloud-wlcb-gpu-077"]="=mlx5_2:1/IB,mlx5_3:1/IB,mlx5_5:1/IB,mlx5_6:1/IB"

input_host=$(hostname)

echo ${host_rank_map[$input_host]}

