
class PolynomialForm:
    def __init__(self, inputs, polynomial):
        self.inputs         = inputs
        self.terms          = tuple(tuple(int(power) for power in term) for term in polynomial)
        self.num_terms      = len(self.terms)
        self.num_var_terms  = sum(any(power > 0 for power in term) for term in self.terms)
        self.order          = max(sum(term) + 1 for term in self.terms)
        assert self.num_terms > 0
        assert all(len(term) == len(self.inputs) for term in self.terms)
        assert all(all(power >= 0 for power in term) for term in self.terms)
        assert len(set(self.terms)) == self.num_terms, 'duplicate terms in polynomial detected'

    def __len__(self):
        return self.num_terms

    def __str__(self):
        terms_list = []
        for term in self.terms:
            parts = []
            for inp, power in zip(self.inputs, term):
                if power > 1:
                    parts.append(f"{inp.name}^{power}")
                elif power == 1:
                    parts.append(f"{inp.name}")
                elif power == 0:
                    pass
            if len(parts) > 1:
                terms_list.append('(' + "*".join(parts) + ')')
            elif len(parts) == 1:
                terms_list.append(parts[0])
            elif len(parts) == 0:
                terms_list.append("1")
        return " + ".join(terms_list)
