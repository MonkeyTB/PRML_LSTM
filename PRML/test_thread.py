#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import vthread

@vthread.pool(6)

def some1(a,b,c):
	import time
	time.sleep(1)
	print("some1",a+b+c)
@vthread.pool(3,1)
def some2(a,b,c):
	import time
	time.sleep(1)
	print("some2",a*b*c)
for i in range(10):
	some1(i,i,i)
	some2(i,i,i)