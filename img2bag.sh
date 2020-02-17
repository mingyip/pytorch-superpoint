#!/bin/bash


for i in "draw_lines" "draw_polygon" "draw_multiple_polygons" "draw_ellipses" "draw_star" "draw_checkerboard" "draw_cube" "draw_stripes"
#for i in "draw_checkerboard" "draw_cube"
# for i in "draw_stripes"
do
	

  for j in "training" "test" "validation"
  do

    if [ "$j" == "training" ]; then
      num=1999
    fi
    if [ "$j" == "test" ]; then
      num=0249
    fi
    if [ "$j" == "validation" ]; then
      num=0099
    fi

    for k in $(eval echo "{0000..$num}")
    do
      echo $j $k
      python scripts/generate_stamps_file.py -i /media/pytorch-superpoint/datasets/$i/images/$j/$k -r 1200.0

      rosrun esim_ros esim_node --data_source=2 --path_to_output_bag=/media/pytorch-superpoint/datasets/$i/images/$j/$k/$k.bag --path_to_data_folder=/media/pytorch-superpoint/datasets/$i/images/$j/$k/ --ros_publisher_frame_rate=60 --exposure_time_ms=10.0 --use_log_image=1 --log_eps=0.1   --contrast_threshold_pos=0.15 --contrast_threshold_neg=0.15
    done


  done


done
