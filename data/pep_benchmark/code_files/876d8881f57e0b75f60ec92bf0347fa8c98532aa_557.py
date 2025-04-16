class book:
    
    
    def __init__(self,bookName,price):
        self.bookName=bookName
        self.price=price
       
        
    
   


a=[]
n=int(input())

for i in range(n):
    b=input()
    c=float(input())
    a.append(book(b,c))
    
max=a[0].price
min=a[0].price
maxi=0
mini=0

for i in range(n):
    
    if(max<a[i].price):
                
        max=a[i].price
        maxi=i
        
            
                
    if(min>a[i].price):
                
        min=a[i].price
        mini=i
        
print('{:.2f},{}'.format(a[maxi].price,a[maxi].bookName))
print('{:.2f},{}'.format(a[mini].price,a[mini].bookName))