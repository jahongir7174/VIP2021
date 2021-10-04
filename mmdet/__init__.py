import mmcv

__version__ = '2.14.0'
short_version = __version__


def parse_version_info(version_str):
    info = []
    for x in version_str.split('.'):
        if x.isdigit():
            info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            info.append(int(patch_version[0]))
            info.append(f'rc{patch_version[1]}')
    return tuple(info)


version_info = parse_version_info(__version__)


def digit_version(version_str):
    info = []
    for x in version_str.split('.'):
        if x.isdigit():
            info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            info.append(int(patch_version[0]) - 1)
            info.append(int(patch_version[1]))
    return info


min_version = '1.3.8'
max_version = '1.4.0'
version = digit_version(mmcv.__version__)

message = f'MMCV=={mmcv.__version__} is incompatible. {min_version}<=MMCV<={max_version}.'
assert (digit_version(min_version) <= version <= digit_version(max_version)), message

__all__ = ['__version__', 'short_version']
