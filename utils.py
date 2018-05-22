import shutil
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
