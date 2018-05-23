import shutil, os, pathlib
def safe_copytree(srcpath, dstpath, *ignores):
    ''' 
    if dst_path doesn't exists,
    it creates new directory and copy all of contents under src
    except *ignores.

    src must exists.
    dst must not exists.

    # ex) copy directory structure without files.
    safe_copytree(src_dir_path, dst_dir_path, '*.*') 
    '''
    try:
        shutil.copytree(srcpath, dstpath, 
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

def make_dstpath(srcpath, old_parent, new_ancestors):
    '''
    discard part of path until old_parent,
    and put new_ancestors into path

    ex) make_dstpath('a/b/cd/dx','cd','123/456') -> '123/456/dx'
    '''
    p = pathlib.Path(srcpath)
    idx = p.parts.index(old_parent)
    return str(pathlib.Path(new_ancestors) \
                      .joinpath(*p.parts[idx+1:]))

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
    print(make_dstpath('./asd/fde/gcd/sdf.jpg','gcd','./aaa/dx/'))
    print(make_dstpath('a/b/cd/dx','cd','123/456'))
    unittest.main()