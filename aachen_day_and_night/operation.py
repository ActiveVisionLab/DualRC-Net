import os
import tarfile
from tqdm import tqdm
import argparse
import shutil

parser = argparse.ArgumentParser(description='operation of aachen day and night')
parser.add_argument('--aachen_path', type=str, default='/home/xinghui/storage/Aachen_Day_Night')
parser.add_argument('--image_pairs', type=str, default='all')
parser.add_argument('--option', type=str, default='')
parser.add_argument('--experiment_name', type=str, default='')
args = parser.parse_args()

if args.image_pairs=='all':
    pair_names_fn = os.path.join(args.aachen_path,'image_pairs_to_match.txt')
elif args.image_pairs=='queries':
    pair_names_fn = os.path.join(args.aachen_path,'query_pairs_to_match.txt')
elif args.image_pairs=='all_v1.1':
    pair_names_fn = os.path.join(args.aachen_path, 'image_pairs_to_match_v1_1.txt')

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def transfer_and_zip():
    db_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/db')
    query_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/query/night/nexus5x/')
    try:
        os.mkdir(os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name, 'db'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name, 'query'))
    except FileExistsError:
        pass

    db_dest = os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name, 'db')
    query_dest = os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name, 'query')

    db_list = os.listdir(db_dir)
    query_list = os.listdir(query_dir)

    for file in tqdm(db_list):
        if args.experiment_name in file:
            file = os.path.join(db_dir, file)
            shutil.copy2(file, db_dest)
    for file in tqdm(query_list):
        if args.experiment_name in file:
            file = os.path.join(query_dir, file)
            shutil.copy2(file, query_dest)


def collect_keypoints():
    image_root = os.path.join(args.aachen_path, 'database_and_query/images_upright')
    dest_root = os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name)

    for dir, subdirs, files in os.walk(image_root):
        for file in files:
            if args.experiment_name in file:
                fn = os.path.join(dir, file)

                # get it parent directories up to 'images_upright'
                parent_dir_list = dir.split('/')
                idx = parent_dir_list.index('images_upright')
                parent_dir_list = parent_dir_list[idx+1:]
                parent_dir = '/'.join(parent_dir_list)

                # create parent dir if not exist
                dest_dir = os.path.join(dest_root, parent_dir)
                try:
                    os.makedirs(dest_dir)
                except FileExistsError:
                    pass
                shutil.copy2(fn, dest_dir)


def distribute_keypoints():
    exp_root = os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name)
    dest_root = os.path.join(args.aachen_path, 'database_and_query/images_upright')

    for dir, subdirs, files in os.walk(exp_root):
        for file in files:
            if args.experiment_name in file:
                fn = os.path.join(dir, file)

                # get it parent directories up to 'images_upright'
                parent_dir_list = dir.split('/')
                idx = parent_dir_list.index('%s_kpts' % args.experiment_name)
                parent_dir_list = parent_dir_list[idx + 1:]
                parent_dir = '/'.join(parent_dir_list)

                # create parent dir if not exist
                dest_dir = os.path.join(dest_root, parent_dir)
                shutil.copy2(fn, dest_dir)


def distribute():
    db_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/db')
    query_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/query/night/nexus5x/')
    file_dir = os.path.join(args.aachen_path, '%s_kpts' % args.experiment_name)
    for f in tqdm(os.listdir(os.path.join(file_dir, 'query'))):
        f = os.path.join(file_dir, 'query', f)
        shutil.copy2(f, query_dir)

    for f in tqdm(os.listdir(os.path.join(file_dir, 'db'))):
        f = os.path.join(file_dir, 'db', f)
        shutil.copy2(f, db_dir)


def delete():
    db_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/db')
    query_dir = os.path.join(args.aachen_path, 'database_and_query/images_upright/query/night/nexus5x/')

    db_list = os.listdir(db_dir)
    query_list = os.listdir(query_dir)

    for file in tqdm(db_list):
        if args.experiment_name in file:
            file = os.path.join(db_dir, file)
            os.remove(file)
            print('removed %s' % file)
    for file in tqdm(query_list):
        if args.experiment_name in file:
            file = os.path.join(query_dir, file)
            os.remove(file)
            print('removed %s' % file)

if args.option == 'collect_keypoints':
    collect_keypoints()
if args.option == 'distribute_keypoints':
    distribute_keypoints()
if args.option == 'delete':
    delete()