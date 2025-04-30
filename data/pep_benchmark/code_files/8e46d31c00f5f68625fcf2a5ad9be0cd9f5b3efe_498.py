from django.contrib import admin

# Register your models here.

from .models import *

import logging
logger = logging.getLogger(__name__)

#admin.site.register(Persona)
#admin.site.register(Operatore)
#admin.site.register(Linea)
#admin.site.register(Nastro)

@admin.register(Linea)
class LineaAdmin(admin.ModelAdmin):
    list_display = ('nome', 'polo', 'descrizione',)
    #fields = [('cognome', 'nome'),'sesso',('data_di_nascita', 'comune_di_nascita', 'paese_di_nascita'),'users']


@admin.register(Persona)
class PersonaAdmin(admin.ModelAdmin):
    list_display = ('cognome', 'nome', 'data_di_nascita', 'comune_di_nascita', 'display_users')
    fields = [('cognome', 'nome'),'sesso',('data_di_nascita', 'comune_di_nascita', 'paese_di_nascita'),'users']

@admin.register(Operatore)
class OperatoreAdmin(admin.ModelAdmin):
    list_display = ('identificativo', 'parametro', 'data_inquadramento', 'persona')
    fields = [('matricola', 'identificativo'),'pin',('parametro', 'data_inquadramento'), 'persona'] 
    search_fields = ['identificativo']

# class NastroInline(admin.TabularInline):
#     model = Nastro

@admin.register(Nastro)
class NastroAdmin(admin.ModelAdmin):    
    list_display = ('foglio', 'ora_inizio', 'polo_monta', 'ora_fine', 'polo_smonta', 'tipologia', 'seguente', 'precedente', )
    fields = [('linea', 'treno'), ('fascia', 'tipologia'), ('ora_inizio', 'polo_monta'), ('ora_fine', 'polo_smonta'), 'seguente', 'precedente', ] 
    list_filter = ('polo_monta', 'tipologia' )
    search_fields = ['linea__nome', 'ora_inizio',]
    # inlines = [
    #     NastroInline,
    # ]
    def get_form(self, request, obj=None, change=False, **kwargs):
        form = super(NastroAdmin, self).get_form(request, obj, **kwargs)
        logger.debug('NastroAdmin get_form obj: %s p: %s s: %s' % (obj, obj.precedente, obj.seguente))
        if obj is not None:
            form.base_fields['seguente'].queryset = Nastro.objects.all().exclude(pk=obj.pk).order_by('ora_inizio')
            form.base_fields['precedente'].queryset = Nastro.objects.all().exclude(pk=obj.pk).order_by('ora_fine')
            # TODO aggiungi filtro per orario più prossimo   
            # form.base_fields['seguente'].queryset = \
            #    Nastro.objects.all().filter(precedente=obj.pk).union(
            #        Nastro.objects.all().exclude(pk=obj.pk).exclude(precedente__isnull = False))#.order_by('ora_inizio')                   
            # form.base_fields['precedente'].queryset = \
            #    Nastro.objects.all().filter(seguente=obj.pk).union(
            #        Nastro.objects.all().exclude(pk=obj.pk).exclude(seguente__isnull = False))#.order_by('ora_inizio')  
            pass
        return form

@admin.register(StatoTurno)
class StatoTurnoAdmin(admin.ModelAdmin):
    list_display = ('stato', 'descrizione')
    
@admin.register(TurnoProgrammato)
class TurnoProgrammatoAdmin(admin.ModelAdmin):
    list_display = ('operatore',  'vettura', 'stato', 'sostituto', 'nastro', 'data', 'nastro_completo',)
    list_filter = ('data', 'nastro__tipologia')
    search_fields = ['operatore__identificativo', 'nastro__linea__nome']
    fields = ('data', 'operatore', 'nastro', 'stato', 'sostituto', 'vettura', 'note')

@admin.register(Vettura)
class VetturaAdmin(admin.ModelAdmin):
    pass
    # list_display = ('data', 'operatore', 'nastro')
    # fields = ('data', 'operatore', 'nastro')

@admin.register(TurnoEffettivo)
class TurnoEffettivoAdmin(admin.ModelAdmin):
    pass
    # list_display = ('data', 'operatore', 'nastro')
    # fields = ('data', 'operatore', 'nastro')

@admin.register(VetturaPerTurno)
class VetturaPerTurnoAdmin(admin.ModelAdmin):
    pass
    # list_display = ('data', 'operatore', 'nastro')
    # fields = ('data', 'operatore', 'nastro')