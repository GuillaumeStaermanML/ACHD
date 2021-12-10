import sys
import os
import numpy
from Cython.Distutils import build_ext
try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
prjdir = os.path.dirname(__file__)


def read(filename):
    return open(os.path.join(prjdir, filename)).read()




extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('version.py').read())
setup(
    name='achd',
    version=__version__,
    author='Guillaume Staerman',
    author_email='guillaume.staerman@telecom-paris.fr',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("libqhull",
                 sources=["./libqhull/geom.c", "./libqhull/geom2.c","./libqhull/global.c", "./libqhull/io.c",  "./libqhull/libqhull.c", "./libqhull/mem.c", "./libqhull/merge.c", "./libqhull/poly.c", "./libqhull/poly2.c", "./libqhull/qset.c", "./libqhull/random.c", "./libqhull/rboxlib.c", "./libqhull/stat.c", "./libqhull/user.c", "./libqhull/usermem.c", "./libqhull/userprintf.c", "./libqhull/userprintf_rbox.c"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-std=c99', '-Wcpp', '-I./libqhull'],
                 language="c"), Extension("achd",
                 sources=["_achd.pyx", "achd.cxx", "qhAdapter.cpp"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-std=c++11', '-Wcpp', '-I./libqhull'],
                 extra_objects=["/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/geom.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/geom2.o","/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/global.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/io.o",  "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/libqhull.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/mem.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/merge.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/poly.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/poly2.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/qset.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/random.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/rboxlib.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/stat.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/user.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/usermem.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/userprintf.o", "/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/build/temp.macosx-10.9-x86_64-3.7/libqhull/userprintf_rbox.o"],
                 language="c++")],
    scripts=[],
    py_modules=['version'],
    packages=[],
    license='License.txt',
    include_package_data=True,
    description='Area of the Convex Hull Depth',
    long_description_content_type='text/markdown',
    url='https://github.com/GuillaumeStaermanML/ACHD',
    download_url='',
    install_requires=["numpy", "cython"],
)
#/Users/staermanguillaume/Desktop/Thèse/Code/ACHD_CPP/libqhull