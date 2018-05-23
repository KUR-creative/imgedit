import shutil, os, pathlib
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

def replace_part_of(srcpath, old_part, new_part):
    '''
    change old_part of srcpath with new_part
    old/new_part must be path or str(not include path delimiters) 
    '''
    p = pathlib.Path(srcpath)
    parts = [part.replace(old_part,new_part) for part in p.parts]
    return pathlib.Path(*parts)


import unittest
class Test_replace_part_of_path(unittest.TestCase):
    def test_replace_part(self):
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(replace_part_of(p,'root','xxxx'),
                         pathlib.Path('xxxx','bbb','ccc','leaf'))           
        self.assertEqual(replace_part_of(p,'leaf','xxxx'),
                         pathlib.Path('root','bbb','ccc','xxxx'))           
        self.assertEqual(replace_part_of(p,'bbb','123'),
                         pathlib.Path('root','123','ccc','leaf'))           

    def test_old_part_is_not_in_srcpath(self):
        # nothing happens.
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(p,replace_part_of(p,'xx','asd'))

if __name__ == '__main__':
    unittest.main()
