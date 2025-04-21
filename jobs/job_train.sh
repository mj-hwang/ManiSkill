python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_larger_train_1000 -e "PickLargerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name pick_smaller_train_1000 -e "PickSmallerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_larger_train_1000 -e "PushLargerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name push_smaller_train_1000 -e "PushSmallerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_larger_train_1000 -e "PlaceLargerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name place_smaller_train_1000 -e "PlaceSmallerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_larger_train_1000 -e "PullLargerCube-v1" -n 1000 -o rgb --only-count-success
python -m mani_skill.examples.motionplanning.panda.run --traj-name pull_smaller_train_1000 -e "PullSmallerCube-v1" -n 1000 -o rgb --only-count-success

python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickLargerCube-v1/motionplanning/pick_larger_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PickSmallerCube-v1/motionplanning/pick_smaller_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushLargerCube-v1/motionplanning/push_larger_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PushSmallerCube-v1/motionplanning/push_smaller_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullLargerCube-v1/motionplanning/pull_larger_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj
python -m mani_skill.trajectory.replay_trajectory --traj-path ./demos/PullSmallerCube-v1/motionplanning/pull_smaller_train_1000.h5 -c pd_ee_delta_pose -o rgb --save-traj