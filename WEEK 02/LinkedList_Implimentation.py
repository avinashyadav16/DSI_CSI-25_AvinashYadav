"""
TASK:
    Implement a Linked List in Python Using OOP and Delete the Nth Node


DESCRIPTION:
    Create a Python program that implements a singly linked list using Object-Oriented Programming (OOP) principles. 
    
    Your implementation should include the following: 
        A Node class to represent each node in the list. 
        A LinkedList class to manage the nodes, with methods to: 
            Add a node to the end of the list Print the list Delete the nth node (where n is a 1-based index) 
            Include exception handling to manage edge cases such as: 
                Deleting a node from an empty list Deleting a node with an index out of range 
    
    Test your implementation with at least one sample list. 


AUTHOR: Avinash Yadav

DATE: 15 June 2025
"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def addNode(self, data):
        newNode = Node(data)

        if not self.head:
            self.head = newNode
        else:
            curr = self.head

            while curr.next:
                curr = curr.next

            curr.next = newNode

    def printList(self):
        curr = self.head

        if not curr:
            print("List is empty.")
            return

        while curr:
            print(curr.data, end=" -> " if curr.next else "\n")
            curr = curr.next

    def delete_nthNode(self, n):
        if not self.head:
            print("Cannot delete from an empty list.")
            return

        if n <= 0:
            print("Index should be a positive integer (1-based).")
            return

        if n == 1:
            print(f"Deleting node at position {n} with value {self.head.data}")
            self.head = self.head.next
            return

        curr = self.head
        prev = None
        count = 1

        while curr and count < n:
            prev = curr
            curr = curr.next
            count += 1

        if not curr:
            print("Index out of range.")
            return

        print(f"Deleting node at position {n} with value {curr.data}")
        prev.next = curr.next


####################################################################################
if __name__ == "__main__":
    ll = LinkedList()
    ll.addNode(4)
    ll.addNode(13)
    ll.addNode(9)
    ll.addNode(25)

    print("Original list:")
    ll.printList()
    print()

    ll.delete_nthNode(3)
    print("After deleting 3rd node:")
    ll.printList()
    print()

    ll.delete_nthNode(1)
    print("After deleting 1st node:")
    ll.printList()
    print()

    ll.delete_nthNode(10)  # Out of range

    ll.delete_nthNode(0)   # Invalid index

    ll.delete_nthNode(2)
    ll.delete_nthNode(1)

    ll.delete_nthNode(1)   # Now list is empty


####################################################################################
"""OUTPUT
Original list:
4 -> 13 -> 9 -> 25

Deleting node at position 3 with value 9
After deleting 3rd node:
4 -> 13 -> 25

Deleting node at position 1 with value 4
After deleting 1st node:
13 -> 25

Index out of range.
Index should be a positive integer (1-based).
Deleting node at position 2 with value 25
Deleting node at position 1 with value 13
Cannot delete from an empty list.
"""
