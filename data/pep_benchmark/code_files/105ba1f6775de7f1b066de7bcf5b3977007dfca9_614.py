#User function Template for python3
'''
	Function to return the value at point of intersection
	in two linked list, connected in y shaped form.
	
	Function Arguments: head_a, head_b (heads of both the lists)
	
	Return Type: value in NODE present at the point of intersection
	             or -1 if no common point.

	Contributed By: Nagendra Jha

	{
		# Node Class
		class Node:
		    def __init__(self, data):   # data -> value stored in node
		        self.data = data
		        self.next = None
	}
'''
def size(node):
    cnt=0
    while node:
        cnt+=1
        node=node.next
    return cnt
def npo(head1,head2,d):
    for i in range(d):
        head1=head1.next
    while head1 and head2:
        if head1==head2:
            return head1.data
        head1=head1.next
        head2=head2.next
    return None
#Function to find intersection point in Y shaped Linked Lists.
def intersetPoint(head1,head2):
    #code here
    diff=size(head1)-size(head2)
    if diff<0: #Second list is bigger
        return npo(head2,head1,diff)
    else:
        return npo(head1,head2,diff)

#Another approach using hasking
#Function to find intersection point in Y shaped Linked Lists.
def intersetPoint(head1,head2):
    nodes=set()
    #code here
    while head1:
        nodes.add(head1)
        head1 = head1.next
 
    # now traverse the second list and find the first node that is
    # already present in the set
    while head2:
        # return the current node if it is found in the set
        if head2 in nodes:
            return head2.data
        head2=head2.next
    # we reach here if lists do not intersect
    return None

#{ 
#  Driver Code Starts
#Initial Template for Python 3

#Contributed by : Nagendra Jha

import atexit
import io
import sys

_INPUT_LINES = sys.stdin.read().splitlines()
input = iter(_INPUT_LINES).__next__
_OUTPUT_BUFFER = io.StringIO()
sys.stdout = _OUTPUT_BUFFER

@atexit.register

def write():
    sys.__stdout__.write(_OUTPUT_BUFFER.getvalue())

# Node Class
class Node:
    def __init__(self, data):   # data -> value stored in node
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        temp=None
    
    # creates a new node with given value and appends it at the end of the linked list
    def append(self, new_node):
        if self.head is None:
            self.head = new_node
            self.temp = self.head
            return
        else:
            self.temp.next = new_node
            self.temp = self.temp.next

if __name__ == '__main__':
    t=int(input())
    for cases in range(t):
        x,y,z = map(int,input().strip().split())
        a = LinkedList()  # create a new linked list 'a'.
        b = LinkedList()  # create a new linked list 'b'.
        nodes_a = list(map(int, input().strip().split()))
        nodes_b = list(map(int, input().strip().split()))
        nodes_common = list(map(int, input().strip().split()))

        for x in nodes_a:
            node=Node(x)
            a.append(node)  # add to the end of the list

        for x in nodes_b:
            node=Node(x)
            b.append(node)  # add to the end of the list

        for i in range(len(nodes_common)):
            node=Node(nodes_common[i])
            a.append(node)  # add to the end of the list a
            if i== 0:
                b.append(node)  # add to the end of the list b, only the intersection
        
        print(intersetPoint(a.head,b.head))

# } Driver Code Ends
