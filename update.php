<!DOCTYPE html>
<html lang="pt-br">
<head>
    <title>Sistema Crud</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="crud">
   <h3>Atualização de Usuário</h3>    
    <?php
     require('../PDWEL-master/Database.php');
     $DataBase = new Database();
     $sql = "UPDATE usuarios SET descricao = :descricao WHERE id = :id";
     $binds = ['descricao'=>'Sou o leandro','id'=>4];
     $result = $DataBase->update($sql, $binds);
     if($result){
     	echo"<div class='sucess'> Atualizado com sucesso </div>";
     }else{
     	echo "Não foi possivel fazer a atualização";
     }
    ?>
</div>
</body>
</html>