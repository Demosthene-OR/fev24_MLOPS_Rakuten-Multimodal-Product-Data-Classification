from unittest.mock import Mock

error = "Veillez entrer des nombres valides!"

def multiplication_of() -> int:
    number_1 = input('Entrez le premier nombre : ')
    number_2 = input('Entrez le deuxième nombre : ')

    try:
        resul = int(number_1) * int(number_2)
    except ValueError:
        return error
    else:
        return resul

def test_multiplication_of():
    with Mock() as mocked_input:
        mocked_input.side_effect = ["4", "3", "d", "5"]
        result = multiplication_of()
        assert isinstance(result, int)
        assert result == 12

        result = multiplication_of()
        assert isinstance(result, str)
        assert result == error

if __name__ == "__main__":
    test_multiplication_of()
