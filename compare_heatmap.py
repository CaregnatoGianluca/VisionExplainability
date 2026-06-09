# %%
import os
import heatmap_similarity_metrices as hsm
import PIL.Image as Image
import pandas as pd
import numpy as np

# The CUB/CXR dataset loaders pull in torch/torchvision and are ONLY needed for
# the CUB/CXR comparison loops. Import them lazily so the KDEF comparison can run
# on a minimal environment (numpy / scipy / pandas / pillow only, no torch).
try:
    from DatasetLoader import cub_v2 as cub
    from DatasetLoader import CXR as cxr
except Exception as e:
    cub = cxr = None
    print(f"[info] CUB/CXR loaders unavailable ({e}). The KDEF comparison still works.")

# %%
# --------------------- EDIT THIS TO CHANGE DATASET AND COMPARISON SETTINGS --------------------- #
DATASET = "kdef"  # Options: "cxr" or "cub" or "kdef"
OUTPUT_TYPE = "gaze"  # Options: "gaze" or "heatmaps"
BASE_WEIGHTS_DIR = "./drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation"


# %%
output_folder = "output_gaze" if OUTPUT_TYPE == "gaze" else "output_heatmaps"

folder_1 = "CNN"
#folder_1 = "Transformer"
folder_2 = "Mamba"
#folder_2 = "Transformer"

model_name_1 = "ResNet50"
cam_name_1 = "GradCAM"
#model_name_1 = "Unfrozen"
#cam_name_1 = "Transformervit_base_patch16_224"

model_name_2 = "Mambavim_base_patch16_224"
cam_name_2 = "original"
#model_name_2 = "Unfrozen"
#cam_name_2 = "Transformervit_base_patch16_224"

# --- KDEF only_gaze folder layout per model (set the matching values above) ---
# The path is: BASE/<folder>/output_gaze/KDEF/<subfolder>/<model_name>/<cam_name>
# Leave a level as "" to skip it.
#   CNN          subfolder="only_gaze"  model_name="ResNet50"                    cam_name="GradCAM"|"ScoreCAM"|"AblationCAM"
#   Mamba        subfolder="only_gaze"  model_name="Mambavim_base_patch16_224"   cam_name="original"
#   Transformer  subfolder=""           model_name="Unfrozen"|"Frozen"           cam_name=""    (KDEF Transformer output is flattened)
# NOTE: for CUB/CXR the Transformer layout is the full one:
#   subfolder="only_gaze"  model_name="Unfrozen"|"Frozen"  cam_name="Transformervit_base_patch16_224"

output_df_name = f"{folder_1}_{folder_2}_{DATASET}_{OUTPUT_TYPE}.csv"


OUTPUT_PATH = "./heatmap_comparison_results"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

output_df_path = os.path.join(OUTPUT_PATH, output_df_name)
print("Output dataframe path:", output_df_path)

subfolders = "only_gaze" if OUTPUT_TYPE == "gaze" else "heatmaps"


# %%
if DATASET == "cub":
    data_folder_name = "CUB_200_2011"
elif DATASET == "cxr":
    data_folder_name = "CXR"
elif DATASET == "kdef":
    data_folder_name = "KDEF"

# %%
#transformer only unfrozen
#cnn only gradcam

# Per-side subfolder level. Defaults to `subfolders` ("only_gaze"/"heatmaps"),
# but can be overridden per model — e.g. the KDEF Transformer output is flattened
# and has no "only_gaze" level, so set subfolders_2 = "" when folder_2 is the
# Transformer for KDEF.
subfolders_1 = subfolders
subfolders_2 = subfolders
#subfolders_2 = ""   # <- uncomment for KDEF Transformer (flattened layout)

complete_path_1 = os.path.join(BASE_WEIGHTS_DIR, folder_1, output_folder, data_folder_name, subfolders_1, model_name_1, cam_name_1)
complete_path_2 = os.path.join(BASE_WEIGHTS_DIR, folder_2, output_folder, data_folder_name, subfolders_2, model_name_2, cam_name_2)

print("Path 1:", complete_path_1)
print("Path 2:", complete_path_2)

# %%
#list file in dir
file_list_1 = os.listdir(complete_path_1)
file_list_2 = os.listdir(complete_path_2)

print(file_list_1)
print(len(file_list_1))

print("\n\n\n")
print(file_list_2)
print(len(file_list_2))

# %%
#create pandas dataframe csv from output if exists, else new
if os.path.exists(os.path.join(OUTPUT_PATH, output_df_name)):
    df_out = pd.read_csv(os.path.join(OUTPUT_PATH, output_df_name))
else:
    df_out = pd.DataFrame()

# %%
DEFAULT_BATCH_SIZE   = 1

# Only needed for the CUB/CXR loops. Skipped automatically on a minimal env
# (where the loaders weren't imported) so the KDEF flow still runs.
if cub is not None and cxr is not None:
    cxr_dataset_options = cxr.dataset_options
    cub_dataset_options = cub.dataset_options

    cub_dataset_options['data_root'] = './drive_folder/Bridging Human and Model Attention_ Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation/CUB/DATASET/'
    cxr_dataset_options['data_root'] = './drive_folder/Bridging Human and Model Attention_ Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation/CXR/'


# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_exp_dataloaders(batchsize=DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], use_padding = False)
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_exp_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'], use_padding = False)
    dataset_options = cub_dataset_options


# %%
if DATASET == "cub":
    df_img = pd.read_csv(os.path.join(cub_dataset_options['data_root'], 'CUB_200_2011', 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
    df_split = pd.read_csv(os.path.join(cub_dataset_options['data_root'], 'CUB_200_2011', 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
    df = pd.concat([df_img, df_split], axis=1)
    df_test = df[df['Train']==0]

# %%
#CUB
if DATASET == "cub":
    results = [] # Temporary list to store new rows

    f1_count = 0
    f2_count = 0

    for img in df_test['Image'].values:
        fname = os.path.basename(img)
        #print(fname)

        if fname not in file_list_1:
            f1_count += 1
            print(f"{fname} not found in {complete_path_1}")
        if fname not in file_list_2:
            f2_count += 1
            print(f"{fname} not found in {complete_path_2}")


        if fname in file_list_1 and fname in file_list_2:
            heatmap_1 = os.path.join(complete_path_1, fname)
            heatmap_2 = os.path.join(complete_path_2, fname)

            print("Comparing:", heatmap_1, "and", heatmap_2)

            # load images
            img_1 = np.asarray(Image.open(heatmap_1))
            img_2 = np.asarray(Image.open(heatmap_2))

            # Calculate scores (Assuming this returns a dictionary)
            # If it returns a list/tuple, you'll need to map them to keys manually
            sim_score = hsm.calc_jss_chi2_pcc_scores(img_1, img_2)

            # 2. Prepare the row
            row = {'filename': fname}

            # If sim_score is a dict like {'jss': 0.1, 'chi2': 0.5...}, update row
            if isinstance(sim_score, dict):
                row.update(sim_score)
            else:
                # If sim_score is a list/tuple, map it manually:
                row['jss'], row['chi2'], row['pcc'] = sim_score

            results.append(row)


    print(f"Files not found in {complete_path_1}: {f1_count}")
    print(f"Files not found in {complete_path_2}: {f2_count}")


    # 3. Combine new results with the existing DataFrame
    if results:
        new_df = pd.DataFrame(results)
        df_out = pd.concat([df_out, new_df], ignore_index=True)

        # 4. Save back to CSV
        df_out.to_csv(output_df_path, index=False)
        print(f"Successfully processed {len(results)} images and saved to {output_df_path}")


# %%
#CXR
if DATASET == "cxr":
    cxr_path = os.path.join(cxr_dataset_options['data_root'], "test")

    results = [] # Temporary list to store new rows

    f1_count = 0
    f2_count = 0
    folders = [folder for folder in os.listdir(cxr_path) if os.path.isdir(os.path.join(cxr_path, folder))]
    for folder in folders:
        folder_path =  os.path.join(cxr_path,folder)

        imgs = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]

        for img in imgs:
            fname = os.path.join(folder, img)


            if img not in file_list_1:
                f1_count += 1
                print(f"{fname} not found in {complete_path_1}")
            if img not in file_list_2:
                f2_count += 1
                print(f"{fname} not found in {complete_path_2}")



            if img in file_list_1 and img in file_list_2:
                heatmap_1 = os.path.join(complete_path_1, img)
                heatmap_2 = os.path.join(complete_path_2, img)

                print("Comparing:", heatmap_1, "and", heatmap_2)

                # load images
                img_1 = np.asarray(Image.open(heatmap_1))
                img_2 = np.asarray(Image.open(heatmap_2))

                # Calculate scores (Assuming this returns a dictionary)
                # If it returns a list/tuple, you'll need to map them to keys manually
                sim_score = hsm.calc_jss_chi2_pcc_scores(img_1, img_2)

                # 2. Prepare the row
                row = {'filename': img}

                # If sim_score is a dict like {'jss': 0.1, 'chi2': 0.5...}, update row
                if isinstance(sim_score, dict):
                    row.update(sim_score)
                else:
                    # If sim_score is a list/tuple, map it manually:
                    row['jss'], row['chi2'], row['pcc'] = sim_score

                results.append(row)


    print(f"Files not found in {complete_path_1}: {f1_count}")
    print(f"Files not found in {complete_path_2}: {f2_count}")


    # 3. Combine new results with the existing DataFrame
    if results:
        new_df = pd.DataFrame(results)
        df_out = pd.concat([df_out, new_df], ignore_index=True)

        # 4. Save back to CSV
        df_out.to_csv(output_df_path, index=False)
        print(f"Successfully processed {len(results)} images and saved to {output_df_path}")


# %%
#KDEF
# Compares the two models' only_gaze (or heatmaps) maps directly from the output
# folders, so no KDEF source data needs to be loaded here.
if DATASET == "kdef":
    results = [] # Temporary list to store new rows

    f1_count = 0
    f2_count = 0

    # KDEF filenames can carry an extra ".jpg" appended to the original ".JPG" name
    # (CNN appends it; Transformer/Mamba keep the original name). Normalize so the
    # same image matches across models regardless of that quirk.
    def _norm_key(name):
        if name.lower().endswith(".jpg") and name[:-4].lower().endswith(".jpg"):
            return name[:-4]
        return name

    map_1 = {_norm_key(f): f for f in file_list_1}
    map_2 = {_norm_key(f): f for f in file_list_2}

    for key in sorted(map_1):
        if key not in map_2:
            f2_count += 1
            print(f"{map_1[key]} not found in {complete_path_2}")
            continue

        heatmap_1 = os.path.join(complete_path_1, map_1[key])
        heatmap_2 = os.path.join(complete_path_2, map_2[key])

        print("Comparing:", heatmap_1, "and", heatmap_2)

        # load images
        img_1 = np.asarray(Image.open(heatmap_1))
        img_2 = np.asarray(Image.open(heatmap_2))

        # Calculate scores (Assuming this returns a dictionary)
        # If it returns a list/tuple, you'll need to map them to keys manually
        sim_score = hsm.calc_jss_chi2_pcc_scores(img_1, img_2)

        # 2. Prepare the row
        row = {'filename': key}

        # If sim_score is a dict like {'jss': 0.1, 'chi2': 0.5...}, update row
        if isinstance(sim_score, dict):
            row.update(sim_score)
        else:
            # If sim_score is a list/tuple, map it manually:
            row['jss'], row['chi2'], row['pcc'] = sim_score

        results.append(row)

    # count files present in folder 2 but missing from folder 1
    for key in map_2:
        if key not in map_1:
            f1_count += 1
            print(f"{map_2[key]} not found in {complete_path_1}")

    print(f"Files not found in {complete_path_1}: {f1_count}")
    print(f"Files not found in {complete_path_2}: {f2_count}")


    # 3. Combine new results with the existing DataFrame
    if results:
        new_df = pd.DataFrame(results)
        df_out = pd.concat([df_out, new_df], ignore_index=True)

        # 4. Save back to CSV
        df_out.to_csv(output_df_path, index=False)
        print(f"Successfully processed {len(results)} images and saved to {output_df_path}")
