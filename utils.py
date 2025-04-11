
class Vec2d:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __add__(self, other):
        if isinstance(other, Vec2d):
            res = Vec2d(self.x + other.x, self.y + other.y)
        elif isinstance(other, (float, int)):
            res = Vec2d(self.x + other, self.y + other)
        else:
            res = Vec2d(-1,-1)
            print("error")
        return res
    
    def __iadd__(self, other):
        if isinstance(other, Vec2d):
            res = Vec2d(self.x + other.x, self.y + other.y)
        elif isinstance(other, (float, int)):
            res = Vec2d(self.x + other, self.y + other)
        else:
            res = Vec2d(-1,-1)
            print("error")
        return res
    
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            res = Vec2d(self.x - other.x, self.y - other.y)
        elif isinstance(other, (float, int)):
            res = Vec2d(self.x - other, self.y - other)
        else:
            res = Vec2d(-1,-1)
            print("error")
        return res
    
    def __isub__(self, other):
        if isinstance(other, Vec2d):
            res = Vec2d(self.x - other.x, self.y - other.y)
        elif isinstance(other, (float, int)):
            res = Vec2d(self.x - other, self.y - other)
        else:
            res = Vec2d(-1,-1)
            print("error")
        return res
    
    def __str__(self):
        return f"Vec2d(x={self.x}, y={self.y})"
    
    def __neg__(self):
        return Vec2d(-self.x, -self.y)
    
    def getTuple(self):
        return (self.x, self.y)