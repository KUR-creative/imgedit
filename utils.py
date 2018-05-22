import shutil, os
def safe_copytree(src, dst, *ignores):
    ''' 
    copy directory tree if dst(arg) path doesn't exists.
    src must exists.
    dst must not exists.

    # ex) copy directory structure without files.
    safe_copytree(src_dir_path, dst_dir_path, '*.*') 
    '''
    try:
        shutil.copytree(src, dst, 
                        ignore=shutil.ignore_patterns(*ignores))
    except Exception as e:
        print(e)

def file_paths(root_dir_path):
    ''' generate file_paths of directory_path ''' 
    it = os.walk(root_dir_path)
    next(it)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name),files):
            yield path
