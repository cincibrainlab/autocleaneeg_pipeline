{% if 'autoclean.mixins' in fullname %}
{{ (fullname.split('.')[-1:] | join('.')) | escape | underline }}
{% else %}
{{ fullname | escape | underline }}
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: bysource
   :no-show-inheritance:
   :no-inherited-members: