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
     $sql = "DELETE FROM usuarios WHERE id = :id";
     $binds = ['id'=>9];
     $result = $DataBase->delete($sql, $binds);
     if($result){
     	echo"<div class='sucess'> Deletado com sucesso </div>";
     }else{
     	echo "Não foi possivel excluir dados";
     }
    ?>
</div>
</body>
</html>