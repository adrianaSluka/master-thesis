# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''

Train YOLOv11. After training, the folder structure is:
```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- detection
                |--- obj_s
                    |--- yolo11-detection-obj_s.pt
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```

------------------------------------------------------    

训练 YOLOv11。训练完成后，文件夹结构如下：
```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- detection
                |--- obj_s
                    |--- yolo11-detection-obj_s.pt
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```
'''

import os
import time
import subprocess

if __name__ == '__main__':

    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    #dataset_path = 'xxx/xxx/demo-bin-picking'
    #dataset_path = '/home/user/Desktop/isaac_project/debug_output_E_hard_negatives'
    '''dataset path includes path to all folder with all 000001-000024 (include hard negatives)'''
    dataset_path = '/xxx/xxx/debug_output_E_hard_negatives'


    
    # Specify the number of GPUs and the number of training epochs.  
    # For example, use 8 GPUs to train for 100 epochs.
    # 指定 GPU 的数量以及训练轮数。  
    # 例如使用 8 张 GPU 进行 100 轮训练。
    gpu_num = 4
    epochs = 50
    
    # Train
    # 开始训练
    dataset_name = os.path.basename(dataset_path)
    task_suffix = 'detection'
    dataset_pbr_path = os.path.join(dataset_path, 'train_pbr')
    train_multi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_train', 'train.py')
    data_objs_path = os.path.join(os.path.dirname(dataset_pbr_path), 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), task_suffix, f"obj_s")
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), save_dir, model_name)
    obj_s_path = os.path.dirname(final_model_path)
    batch_size = 6*gpu_num#12*gpu_num
    start_time = time.time()

    subprocess.run([
        "python", train_multi_path,
        "--data_path", data_objs_path,
        "--epochs", str(epochs),
        "--imgsz", "640",
        "--batch", str(batch_size),
        "--gpu_num", str(gpu_num),
        "--task", task_suffix,
    ], check=True)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Training time: {elapsed/60:.2f} minutes")
    print(f"Training time: {elapsed/3600:.2f} hours")
    pass

