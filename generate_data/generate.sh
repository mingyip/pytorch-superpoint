#!/bin/sh


height=960
width=1280
resize_height=120
resize_width=160
samples=50
frames=60


rm -rf /media/gen/data/

# for draw_foo in "draw_checkerboard" "draw_cube" "draw_ellipses" "draw_lines" "draw_polygon" "draw_star" "draw_stripes" "draw_multiple_polygons"
for draw_foo in "draw_checkerboard" "draw_cube" "draw_lines" "draw_polygon" "draw_star" "draw_stripes" "draw_multiple_polygons"
do
    SECONDS=0
    echo "================== $draw_foo =================="
    echo "===== STEP 1. Generate images and points ====="
    python3 /media/gen/synthetic_dataset.py -d $draw_foo -H $height -W $width -y $resize_height -x $resize_width -i $samples -f $frames --resize

    echo "===== STEP 2. Generate csv files =====" 
    python scripts/generate_stamps_file.py -i /media/gen/data/$draw_foo/images -r 1200.0

    echo "===== STEP 3. Generate bag files ====="
    rosrun esim_ros esim_node \
     --data_source=2 \
     --path_to_output_bag=/media/gen/data/$draw_foo/$draw_foo.bag \
     --path_to_data_folder=/media/gen/data/$draw_foo/images/ \
     --ros_publisher_frame_rate=60 \
     --exposure_time_ms=10.0 \
     --use_log_image=1 \
     --log_eps=0.1 \
     --contrast_threshold_pos=0.15 \
     --contrast_threshold_neg=0.15

    echo "===== STEP 4. Generate event images ====="
    /media/gen/bag2events/build/devel/lib/bag2events/bag2events -b /media/gen/data/$draw_foo/$draw_foo.bag -i /media/gen/data/$draw_foo/events -s $frames

    echo "===== STEP 5. Sanity check ====="
    python3 /media/gen/repack_pts.py -f $frames -d $draw_foo

    # echo "===== STEP 6. (Optional) tar the image folders ====="
    # tar -zcvf /media/gen/data/$draw_foo.tar.gz /media/gen/data/$draw_foo

    duration=$SECONDS
    echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
    echo ""
    echo ""
    echo ""
    echo ""
done

