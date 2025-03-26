{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :exclude-members: __weakref__
   
   {% block methods %}
   .. rubric:: Methods
   
   .. autosummary::
      :nosignatures:
      
      {% for item in methods %}
      {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endfor %}
   {% endblock %}
   
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   
   .. autosummary::
      :nosignatures:
      
      {% for item in attributes %}
      {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endfor %}
   {% endif %}
   {% endblock %} 