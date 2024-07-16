class Test:
	a = 1						### 类变量
	def __init__(self):
		self.member = 2
	def indoor(self, people):
		self.member = people

T = Test()
print(Test.a)			## 使用方法（1）
print(T.a)