import os, sys, tempfile, shutil

class TempDir(object):
    """
    class for temporary directories
    creates a (named) directory which is deleted after use.
    All files created within the directory are destroyed
    Might not work on windows when the files are still opened
    """

    def __init__(self, suffix="", prefix="tmp", basedir=None):
        self.name = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=basedir)

    def __del__(self):
        try:
            if self.name:
                self.dissolve()
        except AttributeError:
            pass

    def __enter__(self):
        return self.name

    def __exit__(self, *errstuff):
        self.dissolve()

    def dissolve(self):
        """remove all files and directories created within the tempdir"""
        if self.name:
            shutil.rmtree(self.name)
        self.name = ""

    def __str__(self):
        if self.name:
            return "temporary directory at: %s" % (self.name,)
        else:
            return "dissolved temporary directory"


class in_tempdir(object):
    """Create a temporary directory and change to it.  """

    def __init__(self, delete_temp=True, *args, **kwargs):
        self.delete_temp = delete_temp
        self.tmpdir = TempDir(*args, **kwargs)

    def __enter__(self):
        self.old_path = os.getcwd()
        os.chdir(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *errstuff):
        os.chdir(self.old_path)
        if self.delete_temp:
            self.tmpdir.dissolve()

class IterableAdapter:
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())