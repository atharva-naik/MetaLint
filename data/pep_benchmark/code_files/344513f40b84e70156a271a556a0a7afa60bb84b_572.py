from odoo import models, fields, api, _
from odoo.exceptions import ValidationError
from datetime import date, datetime
from datetime import datetime, timedelta
class Hr_Holidays_inherited_Model(models.Model):
    _inherit = 'hr.holidays'

    public_holiday=fields.Float(string='Public Holiday In Between',compute='check_public_holiday')

    @api.model
    def create(self, vals):
        holiday_status_id=vals['holiday_status_id']
        # print ("vals date_from",vals['date_from'])
        # print ('state', vals['state'])
        # print ('holiday_status_id is called',holiday_status_id)

        if vals['type'] == 'remove':
            Is_check_hr_holidays_status= self.env['hr.holidays.status'].search([('id','=',holiday_status_id),('exclude_public_holidays','=',True)])
            if Is_check_hr_holidays_status:
                if vals['date_from'] and vals['date_to']:
                    count = 0;

                    start_date = datetime.strptime(vals['date_from'], '%Y-%m-%d %H:%M:%S').date()
                    end_date = datetime.strptime(vals['date_to'], '%Y-%m-%d %H:%M:%S').date()

                    range_of_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
                    for public_holiday_date in range_of_dates:
                        check_public_holidays = self.env['public_holiday.public_holiday'].search([])
                        for pub_holiday in check_public_holidays:
                            if str(public_holiday_date)==pub_holiday.start:
                                count+=1
                            else:
                                pass
                    set_count=vals['number_of_days_temp']-float(count)
                    if vals['number_of_days_temp']<1:
                        vals['number_of_days_temp']=0
                        vals['public_holiday']=0

                    else:
                        vals['number_of_days_temp']=set_count
                        vals['public_holiday'] = float(count)
                    return super(Hr_Holidays_inherited_Model, self).create(vals)
        else:
            return super(Hr_Holidays_inherited_Model, self).create(vals)


    @api.depends('date_from', 'date_to')
    def check_public_holiday(self):
        if self.date_from and self.date_to:
            count = 0;
            start_date = datetime.strptime(self.date_from, '%Y-%m-%d %H:%M:%S').date()
            end_date = datetime.strptime(self.date_to, '%Y-%m-%d %H:%M:%S').date()
            range_of_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
            for public_holiday_date in range_of_dates:
                check_public_holidays = self.env['public_holiday.public_holiday'].search([])
                for pub_holiday in check_public_holidays:
                    if str(public_holiday_date) == pub_holiday.start:
                        count += 1
                    else:
                        pass
                self.public_holiday=count


