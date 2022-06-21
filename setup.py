#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Author: vinowan
# Date: 2021-04-20 12:06:00
# LastEditTime: 2021-04-20 12:06:01
# FilePath: /zmq_ops/setup.py
# Description: 
# 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir + 'zmq_ops']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='zmq_ops',
    version='0.4.0',
    author='Vino Wan',
    author_email='vinowan@tencent.com',
    long_description = readme(),
    long_description_content_type = "text/markdown",
    packages=find_packages(where='.'),
    package_data={"zmq_ops": ["*.so"],},
    ext_modules=[CMakeExtension('zmq_ops')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires = [
        'tensorflow >= 2.3.0'
    ]
)