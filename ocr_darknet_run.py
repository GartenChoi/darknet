import os
import shutil
import logging

IS_PLATFORM=True

class YoloDefaultValues:
    base_root = os.path.abspath(os.path.dirname(__file__))
    def __init__(self, model_name, type='train', is_platform=False):
        # 학습/추론 할 때 사용
        if is_platform:
            platform_root = '/data/aip/common_data/ocr_darknet'
        else:
            platform_root = '/home/ubuntu/jwchoi/ext_lib'
        self.darknet_path = f'{platform_root}/darknet'
        self.custom_names = f'{self.darknet_path}/data/coco.names'
        self.cfg_path = f'{self.darknet_path}/cfg/yolov4.cfg'
        self.model_path = f'{self.base_root}/yolov4.weights'

        # 학습/추론 시 생성
        self.img_tmp_dir = f'{self.base_root}/work/img_tmp'
        self.backup_path = f'{self.base_root}/work/{model_name}/backup'
        self.project_path = f'{self.base_root}/work/{model_name}/custom_data'
        self.detector_path = f'{self.project_path}/obj.data'
        self.train_path=f'{self.project_path}/train.txt'
        self.test_path=f'{self.project_path}/test.txt'
        self.filedata=[]

        if type=='train':
            os.makedirs(self.backup_path, exist_ok=True)
            os.makedirs(self.project_path, exist_ok=True)
        else:
            if not os.path.isdir(self.project_path):
                raise ValueError(f'No Model Exists: {model_name}')
            for model in os.listdir(self.backup_path):
                if model.endswith('_best.weights'):
                    self.model_path = f'{self.backup_path}/{model}'
            idx=0
            inference_path=f'{self.base_root}/work/{model_name}/inference_{idx}'
            while os.path.isdir(inference_path):
                idx+=1
                inference_path = f'{self.base_root}/work/{model_name}/inference_{idx}'
            os.makedirs(inference_path, exist_ok=True)
            self.detect_list_path = f'{inference_path}/detect_list.txt'
            self.detect_result_path = f'{inference_path}/detect_result.txt'

    def get_file_path(self, file_url):
        if '://' not in file_url:
            return file_url
        url_download_prefix='urllib'
        split_list = file_url.split('/')
        local_path = f'{self.img_tmp_dir}/{"/".join(split_list[2:])}'
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            import pyhdfs
            assert not file_url.startswith(url_download_prefix)
            client = pyhdfs.HdfsClient('/'.join(split_list[:3]))
            exists = client.exists(file_url)
            logging.info(exists)
            if not exists:
                raise ValueError(f'file not exists: {file_url}')
            client.copy_to_local(file_url, local_path)
        except:
            import urllib.request
            if file_url.startswith(url_download_prefix):
                file_url=file_url.lstrip(url_download_prefix)
            urllib.request.urlretrieve(file_url, local_path)
        return local_path

def train(tm):
    model_name = tm.param_info.get('model_name', 'default').replace(' ', '_')
    values=YoloDefaultValues(model_name, type='train', is_platform=IS_PLATFORM)
    dataset_root=tm.train_data_path.rstrip("/")
    classes = [line.strip() for line in open(values.custom_names, 'r', encoding='utf-8').readlines() if len(line.strip()) > 0]
    values.file_data = []
    for root, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            label_filename = f"{'.'.join(filename.split('.')[:-1])}.txt"
            if (filename.endswith('.jpg') or filename.endswith('.png')) and os.path.isfile(f'{root}/{label_filename}'):
                values.file_data.append({
                    'image': f'{root}/{filename}',
                    'label': f'{root}/{label_filename}'
                })
            if filename.endswith('.names'):
                values.custom_names=f'{root}/{filename}'
            if filename.endswith('.cfg'):
                values.cfg_path=f'{root}/{filename}'
            if filename.endswith('.weights'):
                values.model_path=f'{root}/{filename}'

    f = open(values.detector_path, 'w', encoding='utf-8')
    f.write(
        f'classes={len(classes)}\n'
        f'train={values.train_path}\n'
        f'valid={values.test_path}\n'
        f'names={values.custom_names}\n'
        f'backup={values.backup_path}'
    )
    f.close()

    train_txt = open(values.train_path, 'w', encoding='utf-8')
    test_txt = open(values.test_path, 'w', encoding='utf-8')
    if len(values.file_data) == 1:
        values.file_data.extend(values.file_data)
    for idx, data in enumerate(values.file_data):
        image_path = data.get('image')
        if idx % 10 == 0:
            test_txt.write(f'{image_path}\n')
        else:
            train_txt.write(f'{image_path}\n')
    train_txt.close()
    test_txt.close()
    os.chdir(values.darknet_path)
    command=f'CUDA_VISIBLE_DEVICES=1 {values.darknet_path}/darknet detector train {values.detector_path} {values.cfg_path} {values.model_path} -dont_show -map'
    os.system(command)

def init_svc(*args, **kwargs):
    pass

def inference(df, params, *args, **kwargs):
    """
        "data" : [
          [model_name],
          [file_path1, file_path2, ... ]
        ]
        "data" : [[file_path1, file_path2, ... ]]
    """
    if len(df) == 1:
        model_name = 'default'
        file_paths = df.values[0].tolist()
    else:
        model_name = df.iloc[0][0]
        file_paths = df.values[1].tolist()
    values=YoloDefaultValues(model_name, type='inference', is_platform=IS_PLATFORM)

    # FILE_DICT={local_path:original_path}
    FILE_DICT={}
    f=open(values.detect_list_path,'w', encoding='utf-8')
    for original_path in file_paths:
        local_path=values.get_file_path(original_path)
        FILE_DICT[local_path]=original_path
        f.write(f'{local_path}\n')
    f.close()

    os.chdir(values.darknet_path)
    command = f'CUDA_VISIBLE_DEVICES=1 {values.darknet_path}/darknet -i 3 detector test {values.detector_path} {values.cfg_path} {values.model_path} -dont_show -ext_output < {values.detect_list_path} > {values.detect_result_path}'
    os.system(command)
    print(f'inference finished. check {values.detect_result_path}')

    f = open(values.detect_result_path, 'r', encoding='utf-8')
    total_dict_list = []
    basic_dict = {}
    file_class_list = []
    for line in f.readlines():
        line=line.strip()
        if '.jpg' in line or '.png' in line:
            if len(file_class_list) > 0:
                basic_dict['object'] = file_class_list
            if len(basic_dict.keys()) > 0:
                total_dict_list.append(basic_dict)
            local_path=line.split(':')[0]
            basic_dict = {'file_tmp_path': local_path, 'file_path': FILE_DICT[local_path]}
            file_class_list = []
        elif '%' in line:
            line = line.replace(')', '').replace('(', '').replace(':', '')
            split_textlist = line.split(' ')
            split_textlist = [tmp for tmp in split_textlist if tmp != '']
            name = split_textlist[0]
            conf = split_textlist[1].split('%')[0]
            coordinates_dict = {'left_x': split_textlist[2], 'top_y': split_textlist[4], 'width': split_textlist[6],
                                'height': split_textlist[-1]}
            one_class_dict = {'class_name': name, 'name': name, 'conf': conf, 'coordinates': coordinates_dict}
            file_class_list.append(one_class_dict)
    if len(basic_dict.keys()) > 0:
        basic_dict['object'] = file_class_list
        total_dict_list.append(basic_dict)
    f.close()
    shutil.rmtree(values.img_tmp_dir)
    return total_dict_list