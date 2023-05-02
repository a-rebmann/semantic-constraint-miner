# generate LTL statements
def neg(x):  # negation helper
    return f'(~{x})'


def release(x, y):  # release helper, as the release operator is not implemented in the LTL solver
    return f'G(~(~{x} U ~{y}))'


def pc_constraints_to_ltl(constraints):  # our custom 'precedes' operator/function
    return [f'G({b} -> {release(neg(b), a)})' for (a, b) in constraints]


# 'precedes-if-occurs' constraints generator
def pio_constraints_to_ltl(constraints):
    return [f'G(F({a}) -> ({b} -> {release(neg(b), a)}))' for (a, b) in constraints]


# 'mutual negation'/NAND constraints generator
def neg_constraints_to_ltl(constraints):
    return [f'G(({a} -> ~{b}) & ({b} -> ~{a}))' for (a, b) in constraints]


# XOR operator/function
def xor_constraints_to_ltl(constraints):
    ltl_statements = []
    for (variables, successors) in constraints:
        ltl_statement_list_1 = []
        for a in variables:
            for b in variables:
                if not a == b:
                    ltl_statement_list_1.append(neg_constraints_to_ltl([(a, b)])[0])
        joined_vars = ' | '.join(variables)
        ltl_statement_list_2 = [f'G({successor} -> F({joined_vars}))' for successor in successors]
        ltl_statements.append('(' + ' & '.join(ltl_statement_list_1) + ' & ' + ' & '.join(ltl_statement_list_2) + ')')
    return ltl_statements


# OR operator/function
def or_constraints_to_ltl(constraints):
    ltl_statements = []
    for (variables, successors) in constraints:
        joined_vars = ' | '.join(variables)
        ltl_statement_list_2 = [f'G({successor} -> F({joined_vars}))' for successor in successors]
        ltl_statements.append('(' + ' & '.join(ltl_statement_list_2) + ')')
    return ltl_statements
