from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import ValidationError, DataRequired

class imgIdForm(FlaskForm):
	imgid=IntegerField("imageID",validators=[DataRequired()])
	submit=SubmitField("Search!")