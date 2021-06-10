from diffir.measure.unsupervised import TopkMeasure


class TestUnsupervisedMeasure:
    def test_tauap_one(self):
        measure = TopkMeasure()
        x = [1, 2, 3]
        y = [1, 2, 3]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == 1

    def test_tauap_two(self):
        measure = TopkMeasure()
        x = [1, 2, 3]
        y = [3, 2, 1]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == -1

    def test_tauap_three(self):
        measure = TopkMeasure()
        x = [1, 2, 5]
        y = [1, 9, 10]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == 1

    def test_tauap_three(self):
        measure = TopkMeasure()
        x = [3, 1, 2]
        y = [1, 2, 3]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == 0

    def test_tauap_four(self):
        measure = TopkMeasure()
        x = [1, 2, 4, 3, 5]
        y = [1, 2, 3, 4, 5]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == 0.75

    def test_tauap_five(self):
        measure = TopkMeasure()
        x = [2, 1, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        tauap = measure.tauap(x, y)
        tauap_fast = measure.tauap_fast(x, y)
        assert tauap == tauap_fast
        assert tauap == 0.875
