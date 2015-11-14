'''
The MIT License (MIT)

Copyright (c) 2012-2013, Alexander Saltanov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

# encoding: utf-8

import errno
import os
import tempfile
import codecs


umask = os.umask(0)
os.umask(umask)


def _maketemp(name, createmode=None):
    """
    Create a temporary file with the filename similar the given ``name``.
    The permission bits are copied from the original file or ``createmode``.

    Returns: the name of the temporary file.
    """
    d, fn = os.path.split(name)
    fd, tempname = tempfile.mkstemp(prefix=".%s-" % fn, dir=d)
    os.close(fd)

    # Temporary files are created with mode 0600, which is usually not
    # what we want. If the original file already exists, just copy its mode.
    # Otherwise, manually obey umask.
    try:
        st_mode = os.lstat(name).st_mode & 0o777
    except OSError as err:
        if err.errno != errno.ENOENT:
            raise
        st_mode = createmode
        if st_mode is None:
            st_mode = ~umask
        st_mode &= 0o666
    os.chmod(tempname, st_mode)

    return tempname


class AtomicFile(object):
    """
    Writeable file object that atomically writes a file.

    All writes will go to a temporary file.
    Call ``close()`` when you are done writing, and AtomicFile will rename
    the temporary copy to the original name, making the changes visible.
    If the object is destroyed without being closed, all your writes are
    discarded.
    If an ``encoding`` argument is specified, codecs.open will be called to open
    the file in the wanted encoding.
    """
    def __init__(self, name, mode="w+b", createmode=None, encoding=None):
        self.__name = name  # permanent name
        self._tempname = _maketemp(name, createmode=createmode)
        if encoding:
            self._fp = codecs.open(self._tempname, mode, encoding)
        else:
            self._fp = open(self._tempname, mode)

        # delegated methods
        self.write = self._fp.write
        self.fileno = self._fp.fileno

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type:
            return
        self.close()

    def close(self):
        if not self._fp.closed:
            self._fp.close()
            os.rename(self._tempname, self.__name)

    def discard(self):
        if not self._fp.closed:
            try:
                os.unlink(self._tempname)
            except OSError:
                pass
            self._fp.close()

    def __del__(self):
        if getattr(self, "_fp", None):  # constructor actually did something
            self.discard()
