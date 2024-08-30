 
from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, args):
    old, new = args.split(',')
    return value.replace(old, new)

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter(name='split')
def split(value, arg):
    return value.split(arg)