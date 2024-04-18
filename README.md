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


<table>
    <tr>
        <td align="center"> <b> Название исследуемой задачи </b> </td>
        <td> Анализ смещения распределений при использовании сравнительного подхода в обучении представлений данных </td>
    </tr>
    <tr>
        <td align="center"> <b> Тип научной работы </b> </td>
        <td> ВКР </td>
    </tr>
    <tr>
        <td align="center"> <b> Автор </b> </td>
        <td> Мария Александровна Никитина </td>
    </tr>
    <tr>
        <td align="center"> <b> Научный руководитель </b> </td>
        <td> кандидат ф.-м. наук, Исаченко Роман Владимирович </td>
    </tr>
</table>

Abstract
========

В последнее время contrastive learning вновь приобрело популярность. Оно заключается в сравнении положительных (похожих) и отрицательных (непохожих) пар из выборки для получения представления без меток. Однако наличие ложноотрицательных и ложноположительных элементов приводит к смещению функции потерь. В данной работе анализируются различные способы устранения этих искажений. Основываясь на примере из области обучения с учителем, разрабатывется несмещённая модель contrastive learning, которая не требует разметки, исследуются её свойства. Качество несмещённого представления оценивается в задаче классификации на примере датасета CIFAR10. В работе демонстрируя применимость и надежность предлагаемого метода в случаях, когда полная разметка данных является дорогостоящей или неосуществимой.

Software modules developed as part of the study
======================================================
1. A python code with all implementation `here <https://github.com/intsystems/Nikitina-BS-Thesis/blob/master/code>`_.
2. A code with base experiment `here <https://github.com/intsystems/Nikitina-BS-Thesis/blob/master/code/Base.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/Nikitina-BS-Thesis/blob/master/code/Base.ipynb>`_.
