#! /usr/bin/env python

# See README.txt for information and build instructions.

from __future__ import print_function
from google.protobuf import text_format
import addressbook_pb2
import sys


# Iterates though all people in the AddressBook and prints info about them.
def EditPeople(address_book):
  for person in address_book.people:
    if person.name == 'Greg Martin II':
      person.name = 'Greg Martin'
    print("Person ID:", person.id)
    print("  Name:", person.name)
    if person.email != "":
      print("  E-mail address:", person.email)

    for phone_number in person.phones:
      if phone_number.type == addressbook_pb2.Person.MOBILE:
        print("  Mobile phone #:", end=" ")
      elif phone_number.type == addressbook_pb2.Person.HOME:
        print("  Home phone #:", end=" ")
      elif phone_number.type == addressbook_pb2.Person.WORK:
        print("  Work phone #:", end=" ")
      print(phone_number.number)
  return address_book

# Iterates though all people in the AddressBook and prints info about them.
def ListPeople(address_book):
  for person in address_book.people:
    print("Person ID:", person.id)
    print("  Name:", person.name)
    if person.email != "":
      print("  E-mail address:", person.email)

    for phone_number in person.phones:
      if phone_number.type == addressbook_pb2.Person.MOBILE:
        print("  Mobile phone #:", end=" ")
      elif phone_number.type == addressbook_pb2.Person.HOME:
        print("  Home phone #:", end=" ")
      elif phone_number.type == addressbook_pb2.Person.WORK:
        print("  Work phone #:", end=" ")
      print(phone_number.number)


# Main procedure:  Reads the entire address book from a file and prints all
#   the information inside.
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "ADDRESS_BOOK_FILE")
  sys.exit(-1)

address_book = addressbook_pb2.AddressBook()

# Read the existing address book.
with open(sys.argv[1], "r") as f:
  # address_book.ParseFromString(f.read())
  text_format.Parse(f.read(), address_book)

address_book = EditPeople(address_book)

# ListPeople(address_book)

with open(sys.argv[1], "w") as f:
  f.write(text_format.MessageToString(address_book))
