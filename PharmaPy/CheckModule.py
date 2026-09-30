# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:49:31 2023

@author: dcasasor
"""

import json
from importlib.resources import files
from typing import Any, Optional

import warnings


def check_modeling_objects(uo: Any, instance_name: Optional[str] = None) -> None:
    """Warn about missing modeling objects using the packaged requirement schema.

    Parameters
    ----------
    uo : object
        Unit operation with an ``oper_mode`` attribute unless its class has a
        special schema entry. Required collaborators depend on mode and class.
    instance_name : str, optional
        Flowsheet instance name included in the diagnostic.

    Warns
    -----
    UserWarning
        If a required collaborator is absent or set to ``None``.

    Raises
    ------
    AttributeError
        If a non-special unit lacks ``oper_mode``.
    KeyError
        If its operation mode is not present in the schema.

    Notes
    -----
    Requirements are package resources, so non-editable installations do not
    depend on a source checkout or the process working directory. This helper
    preserves the existing warning behavior; it does not validate balances.
    """
    resource = files('PharmaPy').joinpath('data/minimum_modeling_objects.json')
    with resource.open('r', encoding='utf-8') as fi:
        checks = json.load(fi)

    class_name = uo.__class__.__name__

    if instance_name is None:
        instance_name = '<InstanceName>'
        instance_descr = "a " + class_name + ' instance'
    else:
        instance_descr = "the '%s' %s instance" % (instance_name, class_name)

    if class_name in checks['special']:
        modeling_objs = checks['special'][class_name]
    else:
        modeling_objs = checks[uo.oper_mode]

    module_name = uo.__module__.split('.')[-1]

    cond_kin = (module_name in checks['has_kinetics']['modules'] or
                class_name in checks['has_kinetics']['classes']) and \
        'Kinetics' not in modeling_objs

    cond_utility = (module_name in checks['has_utility']['modules'] or
                    class_name in checks['has_utility']['classes']) and \
        'Utility' not in modeling_objs

    if cond_kin:
        modeling_objs.append('Kinetics')

    if cond_utility:
        modeling_objs.append('Utility')

    missing_obj = []
    for obj in modeling_objs:
        if not hasattr(uo, obj) or getattr(uo, obj) is None:
            missing_obj.append(obj)

    if len(missing_obj) > 0:
        intro = "The following PharmaPy modeling objects were " \
            "not detected in %s:\n" % instance_descr

        obj_enum = '\t' + ',\n\t'.join(missing_obj) + '.\n\n'

        recommend = "Please create the missing modeling objects listed above" \
            " and then aggregate them one by one to the corresponding unit " \
            "operation instance, e.g. %s.%s = <%sClass>(...)" % (
                    instance_name, missing_obj[0], missing_obj[0])

        message = intro + obj_enum + recommend

        warnings.warn(message)
