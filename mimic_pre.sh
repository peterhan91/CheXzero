python run_preprocess.py \
    --dataset_type mimic \
    --cxr_out_path data/mimic.h5 \
    --csv_out_path data/mimic_paths.csv \
    --mimic_impressions_path data/mimic_impressions.csv \
    --chest_x_ray_path /home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
    --radiology_reports_path /home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/reports/files \
    --resolution 448