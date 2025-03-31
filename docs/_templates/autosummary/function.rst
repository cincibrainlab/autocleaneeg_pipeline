{% if 'autoclean' in fullname %}
{{ (fullname.split('.')[-1:] | join('.')) | escape | underline }}
{% else %}
{{ fullname | escape | underline }}
{% endif %}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

.. _sphx_glr_backreferences_{{ fullname }}:

.. minigallery:: {{ fullname }}
    :add-heading: