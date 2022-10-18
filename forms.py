from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length

class TypeTextForm(FlaskForm):

	file1 = TextAreaField('File 1', validators = [DataRequired()])

	file2 = TextAreaField('File 2', validators = [DataRequired()])

	submit = SubmitField('Text Intersect')