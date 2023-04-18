from pylogics.syntax.base import And, Formula, Implies, Not, Or, _BinaryOp, _UnaryOp, TrueFormula, Logic
from pylogics.syntax.ltl import Always, Eventually, Next, Until, PropositionalTrue, Atomic
from pylogics.utils.to_string import to_string
from pylogics.syntax.pltl import Before, Historically, Once, Since
from pylogics.parsers.ltl import parse_ltl

from semconstmining.declare.enums import Template
from semconstmining.declare.parsers import parse_single_constraint


def to_ltl_str(constraint_str):
    formula = to_ltl(constraint_str)
    return to_string(formula)


def to_ltl(constraint_str):
    n = 0
    constraint = parse_single_constraint(constraint_str)
    templ_str = constraint["template"].templ_str
    if constraint["template"].supports_cardinality:
        n = int(constraint['n'])
    activities = constraint['activities']
    activity_left = Atomic(activities[0].replace(" ", "_"))
    activity_right = None
    if len(activities) == 2:
        activity_right = Atomic(activities[1].replace(" ", "_"))

    if templ_str == Template.ABSENCE.templ_str:
        if n == 1:
            return Not(Eventually(activity_left))
        elif n == 2:
            return Not(Eventually(And(activity_left, Next(Eventually(activity_left)))))
        elif n == 3:
            return Not(Eventually(And(activity_left, Next(Eventually(And(activity_left, Next(Eventually(activity_left))))))))
        elif n == 4:
            return Not(Eventually(And(activity_left, Next(Eventually(And(activity_left, Next(Eventually(And(activity_left, Next(Eventually(activity_left)))))))))))
        else:
            raise ValueError("Unsupported n: " + str(n))

    elif templ_str == Template.EXISTENCE.templ_str:
        if n == 1:
            return Eventually(activity_left)
        elif n == 2:
            return Eventually(And(activity_left, Next(Eventually(activity_left))))
        elif n == 3:
            return Eventually(And(activity_left, Next(Eventually(And(activity_left, Next(Eventually(activity_left)))))))
        else:
            raise ValueError("Unsupported n: " + str(n))

    elif templ_str == Template.EXACTLY.templ_str:
        if n == 1:
            return And(to_ltl(constraint_str.replace(Template.EXACTLY.templ_str, Template.EXISTENCE.templ_str)),
                       to_ltl(constraint_str.replace(Template.EXACTLY.templ_str + "1", Template.ABSENCE.templ_str + "2")))
        elif n == 2:
            return And(to_ltl(constraint_str.replace(Template.EXACTLY.templ_str, Template.EXISTENCE.templ_str)),
                       to_ltl(constraint_str.replace(Template.EXACTLY.templ_str + "2", Template.ABSENCE.templ_str + "3")))
        elif n == 3:
            return And(to_ltl(constraint_str.replace(Template.EXACTLY.templ_str, Template.EXISTENCE.templ_str)),
                       to_ltl(constraint_str.replace(Template.EXACTLY.templ_str + "3", Template.ABSENCE.templ_str + "4")))
        else:
            raise ValueError("Unsupported n: " + str(n))

    elif templ_str == Template.INIT.templ_str:
        return activity_left

    elif templ_str == Template.END.templ_str:
        return Eventually(And(activity_left, Next(Not(PropositionalTrue()))))

    elif templ_str == Template.CHOICE.templ_str:
        return Or(Eventually(activity_left), Eventually(activity_right))

    elif templ_str == Template.EXCLUSIVE_CHOICE.templ_str:
        return And(Or(Eventually(activity_left), Eventually(activity_right)),
                   Not(And(Eventually(activity_left), Eventually(activity_right))))

    elif templ_str == Template.RESPONDED_EXISTENCE.templ_str:
        return Implies(Eventually(activity_left), Eventually(activity_right))

    elif templ_str == Template.RESPONSE.templ_str:
        return Always(Implies(activity_left, Eventually(activity_right)))

    elif templ_str == Template.ALTERNATE_RESPONSE.templ_str:
        return Always(Implies(activity_left, Next(Until(Not(activity_left), activity_right))))

    elif templ_str == Template.CHAIN_RESPONSE.templ_str:
        return Always(Implies(activity_left, Next(activity_right)))

    elif templ_str == Template.PRECEDENCE.templ_str:
        return Or(
            Until(Not(activity_right), activity_left),
            Always(Not(activity_right))
        )
    elif templ_str == Template.ALTERNATE_PRECEDENCE.templ_str:
        return And(
            Or(Until(Not(activity_right), activity_left), Always(Not(activity_right))),
            Always(Implies(activity_right,
                           Or(Until(Not(activity_right), activity_left), Always(Not(activity_right)))
                           )))

    elif templ_str == Template.CHAIN_PRECEDENCE.templ_str:
        return Always(Implies(Next(activity_right), activity_left))

    elif templ_str == Template.SUCCESSION.templ_str:
        return And(to_ltl(constraint_str.replace(Template.SUCCESSION.templ_str, Template.RESPONSE.templ_str)),
                   to_ltl(constraint_str.replace(Template.SUCCESSION.templ_str, Template.PRECEDENCE.templ_str)))

    elif templ_str == Template.ALTERNATE_SUCCESSION.templ_str:
        return And(to_ltl(
            constraint_str.replace(Template.ALTERNATE_SUCCESSION.templ_str, Template.ALTERNATE_RESPONSE.templ_str)),
            to_ltl(constraint_str.replace(Template.ALTERNATE_SUCCESSION.templ_str,
                                          Template.ALTERNATE_PRECEDENCE.templ_str)))

    elif templ_str == Template.CHAIN_SUCCESSION.templ_str:
        return And(
            to_ltl(constraint_str.replace(Template.CHAIN_SUCCESSION.templ_str, Template.CHAIN_RESPONSE.templ_str)),
            to_ltl(constraint_str.replace(Template.CHAIN_SUCCESSION.templ_str, Template.CHAIN_PRECEDENCE.templ_str)))

    elif templ_str == Template.CO_EXISTENCE.templ_str:
        return And(Implies(Eventually(activity_left), Eventually(activity_right)),
                   Implies(Eventually(activity_right), Eventually(activity_left)))

    elif templ_str == Template.NOT_RESPONDED_EXISTENCE.templ_str:
        return Implies(Eventually(activity_left), Not(Eventually(activity_right)))
    elif templ_str == Template.NOT_CHAIN_PRECEDENCE.templ_str:
        return Always(Implies(Next(activity_right), Not(activity_left)))
    elif templ_str == Template.NOT_PRECEDENCE.templ_str:
        return Always(Implies(Eventually(activity_left), Not(activity_left)))
    elif templ_str == Template.NOT_RESPONSE.templ_str:
        return Always(Implies(activity_left, Not(Eventually(activity_right))))
    elif templ_str == Template.NOT_CHAIN_RESPONSE.templ_str:
        return Always(Implies(Next(activity_left), Not(activity_right)))
    elif templ_str == Template.NOT_SUCCESSION.templ_str:
        return And(to_ltl(constraint_str.replace(Template.NOT_SUCCESSION.templ_str, Template.NOT_RESPONSE.templ_str)),
                   to_ltl(constraint_str.replace(Template.NOT_SUCCESSION.templ_str, Template.NOT_PRECEDENCE.templ_str)))
    else:
        raise ValueError("Unknown template: " + constraint["template"].templ_str)
