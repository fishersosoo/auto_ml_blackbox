#!/usr/bin/env python
# -*- coding: utf-8 -*-

class MetaLayer(type):
    def __new__(cls, name, bases, attrs):

        return type.__new__(cls, name, bases, attrs)


class BaseLayer():
    def __init__(self):
        self.layer_type=None
        self.layer_search_space=dict()

    def build_layers(self):
        raise NotImplementedError()


class CNNLayer(BaseLayer):
    layer_type="CNN"
    def __init__(self):

