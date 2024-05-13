from code1 import total
import pytest

def test_total_raises_exception_on_non_list_arguments():
    with pytest.raises(TypeError):
         total(1)

def test_total () :
    #Les use cases :
    """La somme de plusieurs éléments d'une liste doit être correcte"""
    assert(total([1.0, 2.0, 3.0])) == 6.0

    """1 - 1 = 0"""
    assert total([1,-1]) == 0

    """-1 -1 = -2"""
    assert total([-1,-1]) == -2

    #Les edge cases :
    """La somme doit être égal à l'unique élément"""
    assert(total([1.0])) == 1.0

    """La somme d'une liste vide doit être 0"""
    assert total([]) == 0