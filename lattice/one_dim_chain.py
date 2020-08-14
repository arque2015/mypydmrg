"""一维链"""

from . import BaseModel

class HubbardChain(BaseModel):
    """1维Hubbard链\n
    length: 链的长度
    """
    def __init__(self, length, coef_u, pbc=True):
        super().__init__(list(range(1, length+1)))
        self._length = length
        self._pbc = pbc
        self._coef_u = coef_u
        for idx in range(2, length):
            self._bonds[idx] = [idx-1, idx+1]
        if pbc:
            self._bonds[1] = [length, 2]
            self._bonds[length] = [length-1, 1]
        else:
            self._bonds[1] = [2]
            self._bonds[length] = [length-1]
