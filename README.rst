|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: "Анализ смещения распределений при использовании сравнительного подхода в обучении представлений данных"
    :Тип научной работы: НИР
    :Автор: Мария Александровна Никитина
    :Научный руководитель: кандидат ф.-м. наук, Исаченко Роман Вячеславович

Abstract
========

В последнее время contrastive learning вновь приобрело популярность. Оно заключается в сравнении положительных (похожих) и отрицательных (непохожих) пар из выборки для получения представления без меток. Однако наличие ложноотрицательных и ложноположительных элементов приводит к смещению функции потерь. В данной работе анализируются различные способы устранения этих искажений. Основываясь на примере из области обучения с учителем, разрабатывется несмещённая модель contrastive learning, которая не требует разметки, исследуются её свойства. Качество несмещённого представления оценивается в задаче классификации на примере датасета CIFAR10. В работе демонстрируя применимость и надежность предлагаемого метода в случаях, когда полная разметка данных является дорогостоящей или неосуществимой.

Software modules developed as part of the study
======================================================
1. A python code with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/code>`_.
2. A code with base experiment `here <https://github.comintsystems/ProjectTemplate/blob/master/code/base.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/base.ipynb>`_.
