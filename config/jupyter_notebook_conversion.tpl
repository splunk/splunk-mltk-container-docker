{% extends 'python.tpl'%}
{% block markdowncell -%}
{% endblock markdowncell %}
{% block codecell %}
{% if 'name' in cell['metadata'] %}
    {{ super() }}
{% endif %}
{% endblock codecell %}
