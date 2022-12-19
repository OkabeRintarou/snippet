#!/usr/bin/env python3

import sys
import os

def write_file(file, content):
    with open(file, 'w+') as f:
        f.write(content)

def build_sh_content(name, version):
    m = '{}-{}'.format(name, version)
    content='sudo rm -rf /usr/src/{}\n' \
        'sudo cp -r ../{} /usr/src/{}\n' \
        'sudo dkms remove -m {} -v {} --all --force\n' \
        'sudo dkms install -m {} -v {} --force\n'.format(
        m, m, m, name, version, name, version)
    return content

def dkms_conf_content(name, version):
    content='PACKAGE_NAME="{}"\n' \
        'PACKAGE_VERSION="{}"\n' \
        'CLEAN="make clean"\n' \
        'MAKE[0]="make all KVERSION=$kernelver"\n'\
        'BUILT_MODULE_NAME[0]="{}"\n' \
        'DEST_MODULE_LOCATION[0]="/updates"\n' \
        'AUTOINSTALL="yes"\n'.format(name, version, name)
    return content

def make_file_content(name):
    content='obj-m:={}.o\n'\
        'KVERSION=$(shell uname -r)\n\n' \
        'all:\n' \
        '\t$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules\n'\
        'clean:\n'\
        '\t$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean\n'.format(name)
    return content

def source_content(name):
    c='#include <linux/kernel.h>\n' \
        '#include <linux/init.h>\n' \
        '#include <linux/module.h>\n\n' \
        'static int __init %s_init(void)\n{\n' \
        '\treturn 0;\n}\n\n' \
        'static void __exit %s_exit(void)\n{\n}\n\n' \
        'module_init(%s_init);\n' \
        'module_exit(%s_exit);\n\n' \
        'MODULE_LICENSE("GPL v2");\n' \
        'MODULE_AUTHOR("syl");\n' % (name, name, name, name)
    return c

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: %s <module-name> <module-version>' % (sys.argv[0],))
        print('\texample: %s hello 0.1' % (sys.argv[0],))
        sys.exit(-1)

    module_name, module_version = sys.argv[1], sys.argv[2]

    module_dir = './{}-{}'.format(module_name, module_version)
    if os.path.exists(module_dir):
        print('{} already exists'.format(module_dir))
        sys.exit(-1)

    try:
        os.makedirs(module_dir)
        write_file('{}/build.sh'.format(module_dir), build_sh_content(module_name, module_version))
        write_file('{}/dkms.conf'.format(module_dir), dkms_conf_content(module_name, module_version))
        write_file('{}/Makefile'.format(module_dir), make_file_content(module_name))
        write_file('{}/{}.c'.format(module_dir, module_name), source_content(module_name))
        print('Success')
    except Exception as e:
        print(e)
