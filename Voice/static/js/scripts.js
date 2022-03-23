$("form[name=signup_form]").submit(function(e)
{
  var $form=$(this);
var $error= $form.find(".error");
$.ajax({url:"/user/signup",type:"POST",data:data,dataType:"json",
sucess: function(resp)
{
  console.log(resp);
},
error: function(resp)
{
  console.log(resp);
    $error.text(resp.responseJSON.error)
}

});
var data=$form.serialize();
  e.preventDefault();
});
