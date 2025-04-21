python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_larger_validation_100 -e "PickLargerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_smaller_validation_100 -e "PickSmallerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_larger_validation_100 -e "PushLargerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_smaller_validation_100 -e "PushSmallerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_larger_validation_100 -e "PlaceLargerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_smaller_validation_100 -e "PlaceSmallerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_larger_validation_100 -e "PullLargerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_smaller_validation_100 -e "PullSmallerCube-v1" -n 100 -o rgb --only-count-success --start-seed 1000

python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_larger_validation_100 -e "PickLargerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_smaller_validation_100 -e "PickSmallerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_larger_validation_100 -e "PushLargerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_smaller_validation_100 -e "PushSmallerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_larger_validation_100 -e "PlaceLargerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_smaller_validation_100 -e "PlaceSmallerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_larger_validation_100 -e "PullLargerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_smaller_validation_100 -e "PullSmallerCubeSwapped-v1" -n 100 -o rgb --only-count-success --start-seed 2000

python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickLargerCube-v1/motionplanning/pick_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickSmallerCube-v1/motionplanning/pick_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushLargerCube-v1/motionplanning/push_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushSmallerCube-v1/motionplanning/push_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceLargerCube-v1/motionplanning/place_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullLargerCube-v1/motionplanning/pull_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullSmallerCube-v1/motionplanning/pull_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj

python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickLargerCubeSwapped-v1/motionplanning/pick_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickSmallerCubeSwapped-v1/motionplanning/pick_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushLargerCubeSwapped-v1/motionplanning/push_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushSmallerCubeSwapped-v1/motionplanning/push_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceLargerCubeSwapped-v1/motionplanning/place_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceSmallerCubeSwapped-v1/motionplanning/place_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullLargerCubeSwapped-v1/motionplanning/pull_larger_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullSmallerCubeSwapped-v1/motionplanning/pull_smaller_validation_100.h5 -c pd_ee_delta_pose -o rgb --save-traj
