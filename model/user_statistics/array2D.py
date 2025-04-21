
#Created by ChatGPT 4o

class Array2D:
    def __init__(self):
        self.data = []

    def clear(self):
        self.data = []

    def add_row(self, row):
        """添加一整行"""
        self.data.append(row)

    def add_element(self, row_index, value):
        """向指定行添加一个元素"""
        while len(self.data) <= row_index:
            self.data.append([])
        self.data[row_index].append(value)

    def set(self, row, col, value):
        """设置指定位置的值，如果超出则自动扩展"""
        while len(self.data) <= row:
            self.data.append([])

        while len(self.data[row]) <= col:
            self.data[row].append(0)

        self.data[row][col] = value


    def add(self, row, col, addvalue):
        """设置指定位置的值，如果超出则自动扩展"""
        while len(self.data) <= row:
            self.data.append([])

        while len(self.data[row]) <= col:
            self.data[row].append(0)

        self.data[row][col] += addvalue

    def get(self, row, col):
        """获取指定位置的值（如果越界则返回 None）"""
        if row < len(self.data) and col < len(self.data[row]):
            return self.data[row][col]
        return None

    def delete(self, row, col):
        """删除指定位置的元素（设为 0）"""
        if row < len(self.data) and col < len(self.data[row]):
            self.data[row][col] = 0

    def __str__(self):
        return '\n'.join(str(row) for row in self.data)
