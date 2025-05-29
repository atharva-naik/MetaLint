# -*- coding: utf-8 -*-

from odoo import models, fields, api

class SaleWizard(models.TransientModel):
    
    _name = 'academy.sale.wizard'
    _description = 'Wizard: Rapida orden de venta para estudiantes de session'
    
    
    def _default_session(self):
        return self.env['academy.session'].browse(self._context.get('active_id'))
    
    session_id = fields.Many2one(comodel_name='academy.session',
                                string='Session',
                                required=True,
                                default= _default_session)
    
    session_student_ids = fields.Many2many(comodel_name='res.partner',
                                          string='Students in the current session',
                                          related='session_id.students_ids',
                                          help='these are the students currently in the session')
    
    student_ids = fields.Many2many(comodel_name='res.partner', string='students for sale order')
    
    def create_sale_order(self):
        session_product_id = self.env['product.product'].search([('is_ssesion_product','=',True)], limit=1)
    
        if session_product_id:
            for student in self.student_ids:
                order_id = self.env['sale.order'].create({
                    'partner_id': student.id,
                    'session_id': self.session_id.id,
                    'order_line': [(0,0,{'product_id':session_product_id.id, 'price_unit':self.session_id.total_price})]
                })
                
                
                
                
                
        
    
    